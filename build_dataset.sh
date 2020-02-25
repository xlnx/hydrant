#!/bin/sh

raw_path=$1
dst_dir=$2

toolkit_dir=./build/external_build/varch/tools
tmp_dir=/tmp/downsample

dx=256
dy=256
dz=256

log_bs=6
pad=2

lvls=1

raw_base=$(basename ${raw_path})

echo "building ${dst_dir}..."

mkdir -p ${tmp_dir}
mkdir -p ${dst_dir}

lvl_arch() {
    lvl_raw_path=$1
    lvl_raw_base=$(basename ${lvl_raw_path})

    ${toolkit_dir}/archiver \
            -o ${dst_dir} \
            -i ${lvl_raw_path} \
            -x $2 -y $3 -z $4 \
            -p ${pad} -s ${log_bs} \
        >> ${tmp_dir}/archiver.log

    ${toolkit_dir}/archive-thumbnailer \
            -o ${dst_dir} \
            -i ${dst_dir}/${lvl_raw_base%.*}_$((1 << ${log_bs}))p${pad}.h264 \
            -x chebyshev \
        >> ${tmp_dir}/archive-thumbnailer.log
}

# set -x

lvl_arch ${raw_path} $dx $dy $dz

for i in $(seq ${lvls})
do
    echo "sampling level $i/${lvls}..."

    ${toolkit_dir}/downsampler -i ${raw_path} -x $dx -y $dy -z $dz -s $i -o ${tmp_dir}\
        >> ${tmp_dir}/downsampler.log

    lvl_raw_path=${tmp_dir}/${raw_base%.*}_x$((1 << $i)).raw

    lvl_arch ${lvl_raw_path} $(($dx >> i)) $(($dy >> i)) $(($dz >> i))
done

echo "writing package meta..."

${toolkit_dir}/meta-builder \
        -d ${dst_dir} \
    >> ${tmp_dir}/meta-builder.log

# set +x

echo "built ${dst_dir}"
