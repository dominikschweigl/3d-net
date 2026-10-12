FROM swarm.azurecr.io/hailo-ai-sw-suite:2025.4

WORKDIR /

COPY ./converter/* .

RUN pip install -r requirements.txt && \
    pip uninstall nvidia-cudnn-cu11==8.5.0.96 -y

USER root
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]