#!/usr/bin/env bash

docker run --rm coursegen:fixed --theory-per-request 10 --calc-per-request 10 --request-delay 2 --temperature 0.7
