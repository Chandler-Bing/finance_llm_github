#!/bin/bash

# This command wrapper sets up SSH config files based on the following
# environment variables:
#
#   SSH_CONFIG - contents of an SSH config file
#   SSH_KNOWN_HOSTS - contents of a SSH known_hosts file
#   SSH_PRIVATE_RSA_KEY - contents of a SSH private RSA key
#   SSH_PRIVATE_DSA_KEY - contents of a SSH private DSA key
#   SSH_DEBUG - switch to a high debug level 3 for all hosts, to help solve SSH issues
#
# The environment variables are unset after the files are created to help
# prevent accidental output in logs

set -e

if [ -z "$SSH_CONFIG" ] && \
  [ -z "$SSH_CONFIG_B64" ] && \
  [ -z "$SSH_CONFIG_PATH" ] && \
  [ -z "$SSH_KNOWN_HOSTS" ] && \
  [ -z "$SSH_KNOWN_HOSTS_B64" ] && \
  [ -z "$SSH_KNOWN_HOSTS_PATH" ] && \
  [ -z "$SSH_PRIVATE_RSA_KEY" ] && \
  [ -z "$SSH_PRIVATE_RSA_KEY_B64" ] && \
  [ -z "$SSH_PRIVATE_RSA_KEY_PATH" ] && \
  [ -z "$SSH_PRIVATE_DSA_KEY" ] && \
  [ -z "$SSH_PRIVATE_DSA_KEY_B64" ] && \
  [ -z "$SSH_PRIVATE_DSA_KEY_PATH" ] && \
  [ -z "$SSH_DEBUG" ]; then
    # none of the ENV vars we care about found, so skip the logic in this script
    [[ $1 ]] && exec "$@"
fi

mkdir -p /root./ssh
chmod 700 /root./ssh

decode_base64() {
  # Determine the platform dependent base64 decode argument
  if [ "$(echo 'eA==' | base64 -d 2> /dev/null)" = 'x' ]; then
    local BASE64_DECODE_ARG='-d'
  else
    local BASE64_DECODE_ARG='--decode'
  fi

  echo "$1" | tr -d '\n' | base64 "$BASE64_DECODE_ARG"
}

## /root./ssh/config

[[ ! -z "$SSH_CONFIG" ]] && \
  echo "$SSH_CONFIG" > /root./ssh/config && \
  chmod 600 /root./ssh/config && \
  unset SSH_CONFIG

[[ ! -z "$SSH_CONFIG_B64" ]] && \
  decode_base64 "$SSH_CONFIG_B64" > /root./ssh/config && \
  chmod 600 /root./ssh/config && \
  unset SSH_CONFIG_B64

[[ ! -z "$SSH_CONFIG_PATH" && ! -a /root./ssh/config ]] && \
  cp "$SSH_CONFIG_PATH" /root./ssh/config && \
  chmod 600 /root./ssh/config && \
  unset SSH_CONFIG_PATH

## /root./ssh/known_hosts

[[ ! -z "$SSH_KNOWN_HOSTS" ]] && \
  echo "$SSH_KNOWN_HOSTS" > /root./ssh/known_hosts && \
  chmod 600 /root./ssh/known_hosts && \
  unset SSH_KNOWN_HOSTS

[[ ! -z "$SSH_KNOWN_HOSTS_B64" ]] && \
  decode_base64 "$SSH_KNOWN_HOSTS_B64" > /root./ssh/known_hosts && \
  chmod 600 /root./ssh/known_hosts && \
  unset SSH_KNOWN_HOSTS_B64

[[ ! -z "$SSH_KNOWN_HOSTS_PATH" && ! -a /root./ssh/known_hosts ]] && \
  cp "$SSH_KNOWN_HOSTS_PATH" /root./ssh/known_hosts && \
  chmod 600 /root./ssh/known_hosts && \
  unset SSH_KNOWN_HOSTS_PATH

## /root./ssh/id_rsa

[[ ! -z "$SSH_PRIVATE_RSA_KEY" ]] && \
  echo "$SSH_PRIVATE_RSA_KEY" > /root./ssh/id_rsa && \
  chmod 600 /root./ssh/id_rsa && \
  unset SSH_PRIVATE_RSA_KEY

[[ ! -z "$SSH_PRIVATE_RSA_KEY_B64" ]] && \
  decode_base64 "$SSH_PRIVATE_RSA_KEY_B64" > /root./ssh/id_rsa && \
  chmod 600 /root./ssh/id_rsa && \
  unset SSH_PRIVATE_RSA_KEY_B64

[[ ! -z "$SSH_PRIVATE_RSA_KEY_PATH" && ! -a /root./ssh/id_rsa ]] && \
  cp "$SSH_PRIVATE_RSA_KEY_PATH" /root./ssh/id_rsa && \
  chmod 600 /root./ssh/id_rsa && \
  unset SSH_PRIVATE_RSA_KEY_PATH

## /root./ssh/id_dsa

[[ ! -z "$SSH_PRIVATE_DSA_KEY" ]] && \
  echo "$SSH_PRIVATE_DSA_KEY" > /root./ssh/id_dsa && \
  chmod 600 /root./ssh/id_dsa && \
  unset SSH_PRIVATE_DSA_KEY

[[ ! -z "$SSH_PRIVATE_DSA_KEY_B64" ]] && \
  decode_base64 "$SSH_PRIVATE_DSA_KEY_B64" > /root./ssh/id_dsa && \
  chmod 600 /root./ssh/id_dsa && \
  unset SSH_PRIVATE_DSA_KEY_B64

[[ ! -z "$SSH_PRIVATE_DSA_KEY_PATH" && ! -a /root./ssh/id_dsa ]] && \
  cp "$SSH_PRIVATE_DSA_KEY_PATH" /root./ssh/id_dsa && \
  chmod 600 /root./ssh/id_dsa && \
  unset SSH_PRIVATE_DSA_KEY_PATH

## ssh debug mode

[[ ! -z "$SSH_DEBUG" ]] && \
  touch /root./ssh/config && \
  chmod 600 /root./ssh/config && \
  echo -e "Host *\n  LogLevel DEBUG3" >> /root./ssh/config && \
  unset SSH_DEBUG

[[ $1 ]] && exec "$@"
