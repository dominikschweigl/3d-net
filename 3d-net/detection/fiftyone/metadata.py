import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import struct
from urllib.parse import urlparse

import boto3
import cv2
import fiftyone as fo
import numpy as np


def _get_s3_client(endpoint_url: str, region_name: str):
	return boto3.client(
		"s3",
		endpoint_url=endpoint_url,
		aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
		aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
		region_name=region_name,
	)


def _parse_bucket_key_from_url(url: str) -> tuple[str, str]:
	parsed = urlparse(url)
	path = parsed.path.lstrip("/")
	if "/" not in path:
		raise ValueError(f"Cannot parse bucket/key from URL: {url}")
	bucket, key = path.split("/", 1)
	return bucket, key


def _dims_from_header(header: bytes, ext: str) -> tuple[int, int] | None:
	if ext == ".png":
		if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
			w = struct.unpack(">I", header[16:20])[0]
			h = struct.unpack(">I", header[20:24])[0]
			return w, h
		return None

	# JPEG markers
	i = 2
	while i + 4 <= len(header):
		if header[i] != 0xFF:
			break
		marker = header[i + 1]
		if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
			if i + 9 <= len(header):
				h = struct.unpack(">H", header[i + 5 : i + 7])[0]
				w = struct.unpack(">H", header[i + 7 : i + 9])[0]
				return w, h
		length = struct.unpack(">H", header[i + 2 : i + 4])[0]
		i += 2 + length

	return None


def _fetch_image_metadata(s3_client, bucket: str, key: str) -> fo.ImageMetadata:
	head = s3_client.head_object(Bucket=bucket, Key=key)
	size_bytes = head.get("ContentLength")
	mime_type = head.get("ContentType")

	ext = Path(key).suffix.lower()
	resp = s3_client.get_object(Bucket=bucket, Key=key, Range="bytes=0-4095")
	header = resp["Body"].read()
	dims = _dims_from_header(header, ext)

	if dims is None:
		resp = s3_client.get_object(Bucket=bucket, Key=key)
		data = np.frombuffer(resp["Body"].read(), dtype=np.uint8)
		img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
		if img is None:
			raise ValueError(f"Failed to decode image: s3://{bucket}/{key}")
		h, w = img.shape[:2]
		num_channels = 1 if len(img.shape) == 2 else img.shape[2]
	else:
		w, h = dims
		num_channels = None

	return fo.ImageMetadata(
		size_bytes=size_bytes,
		mime_type=mime_type,
		width=w,
		height=h,
		num_channels=num_channels,
	)


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Compute FiftyOne metadata for S3-backed samples")
	parser.add_argument("--dataset", default="m3fd_rgbt", help="Dataset name (default: %(default)s)")
	parser.add_argument("--s3-endpoint-url", default="https://s3.de-west-1.psmanaged.com")
	parser.add_argument("--s3-region", default="de-west-1")
	parser.add_argument("--max-workers", type=int, default=16)
	parser.add_argument("--max-failure-logs", type=int, default=20)
	return parser.parse_args()


def main() -> None:
	args = _parse_args()
	dataset = fo.load_dataset(args.dataset)
	s3_client = _get_s3_client(args.s3_endpoint_url, args.s3_region)

	count_before = dataset.count("metadata")
	
	missing_view = dataset.match(fo.ViewField("metadata").exists(False))
	missing_samples = list(missing_view.iter_samples(progress=True))
	print("samples with metadata:", count_before)
	print("missing:", len(missing_samples))
	print("total:", len(dataset))

	if missing_samples:
		print("Computing metadata for remaining samples...")

	saved = 0
	failed = 0

	with dataset.save_context() as context:
		with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
			futures = {}
			for sample in missing_samples:
				filepath = sample.filepath
				if not isinstance(filepath, str) or not filepath.startswith(("http://", "https://")):
					failed += 1
					continue

				future = executor.submit(
					lambda path: _fetch_image_metadata(s3_client, *_parse_bucket_key_from_url(path)),
					filepath,
				)
				futures[future] = sample

			for future in as_completed(futures):
				sample = futures[future]
				try:
					sample.metadata = future.result()
					context.save(sample)
					saved += 1
				except Exception as exc:
					if failed < args.max_failure_logs:
						print(f"failed: {sample.id} ({sample.filepath}) -> {type(exc).__name__}: {exc}")
					failed += 1

	if failed > args.max_failure_logs:
		print(f"... plus {failed - args.max_failure_logs} additional fallback failures")

	dataset.save()

	count_after = dataset.count("metadata")
	print("Saved Metadata:", saved)
	print("Failed Metadata:", failed)
	print("samples with metadata after:", count_after)
	print("total:", len(dataset))


if __name__ == "__main__":
	main()