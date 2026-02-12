FROM alpine:3.23 AS build
ADD . /usr/src/chamois
RUN apk add --no-cache --force-non-repository -t build-dependencies musl-dev git rust cargo
RUN apk add --no-cache --force-non-repository -t run-dependencies python3 python3-dev py3-setuptools py3-pip py3-wheel py3-zstandard py3-rich py3-psutil py3-h5py py3-numpy py3-scipy py3-pandas
RUN env python3 -m pip install -v --break-system-packages gb-io pyhmmer pyrodigal
RUN env python3 -m pip wheel -v -w /tmp --no-build-isolation --no-deps gb-io pyhmmer pyrodigal /usr/src/chamois

FROM alpine:3.23 AS run
COPY --from=0 /tmp/*.whl /tmp/
RUN apk add --no-cache --force-non-repository -t run-dependencies python3 python3-dev py3-setuptools py3-pip py3-wheel py3-numpy py3-scipy py3-h5py py3-pandas py3-zstandard py3-rich py3-psutil
RUN python3 -m pip install chamois-tool --no-cache-dir --only-binary :all: --find-links /tmp --break-system-packages

ENV HDF5_DISABLE_VERSION_CHECK=1
VOLUME "/io"
WORKDIR "/io"
ENTRYPOINT ["chamois"]
CMD ["--help"]
