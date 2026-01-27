FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel as builder
#FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
#FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
# https://hub.docker.com/r/pytorch/pytorch


LABEL org.opencontainers.image.authors="xinzhimin6@163.com"
LABEL com.example.version="flash-attn"
WORKDIR /app

COPY ./requirements.txt ./

RUN apt-get update && apt-get install -y
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone && \
    apt -y install vim wget curl git unzip libopenmpi-dev openmpi-bin && \
    sed -i '$a\set enc=utf8' /etc/vim/vimrc && \
    apt -y --no-install-recommends install openssh-server openssh-client pdsh ninja-build \
        iputils-ping net-tools rsync dnsutils


RUN pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple && \
pip config set global.trusted-host pypi.mirrors.ustc.edu.cn && \
pip install -r ./requirements.txt

    
##############################################################################
# SSH Config
##############################################################################
ARG SSH_PORT=22
COPY ./config/ssh-env-config.sh /usr/local/bin/ssh-env-config.sh
RUN chmod +x /usr/local/bin/ssh-env-config.sh && \
    echo 'root:NdjxS+-6gEPcq}D' | chpasswd && \
    echo "ClientAliveInterval 30" >> /etc/ssh/sshd_config && \
    sed "0,/^#Port 22/s//Port ${SSH_PORT}/"  /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config && \
    printf "StrictHostKeyChecking no\nUserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    mkdir -p /root/.ssh && \
    cd /root/.ssh && \
    ssh-keygen -t rsa -f /root/.ssh/id_rsa -N "" && cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys && \
    chmod og-wx /root/.ssh/authorized_keys
EXPOSE ${SSH_PORT}

# 启动容器时，启动ssh服务，并config
# CMD /etc/init.d/ssh start && ssh-env-config.sh /bin/bash
