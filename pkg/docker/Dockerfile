FROM alpine:3.21.3 as build
ADD . /usr/src/chamois
RUN apk add --no-cache alpine-sdk musl-dev linux-headers rust cargo python3 python3-dev py3-pip hdf5-dev
RUN env python3 -m pip wheel /usr/src/chamois --wheel-dir /tmp -v # --no-build-isolation

FROM alpine:3.21.3 as run
COPY --from=0 /tmp/*.whl /tmp/
RUN apk add --no-cache python3 py3-pip py3-wheel py3-numpy py3-scipy py3-h5py py3-pandas py3-biopython py3-psutil py3-lz4
RUN python3 -m pip install chamois --no-index --only-binary :all: --find-links /tmp --break-system-packages

ENV HDF5_DISABLE_VERSION_CHECK=1
VOLUME "/io"
WORKDIR "/io"
ENTRYPOINT ["chamois"]
CMD ["--help"]
