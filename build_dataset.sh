#!/bin/bash

OPTIND=1

raw_path=""
dst_dir=./ds
log_bs=6
pad=2
lvls=3

toolkit_dir=./bin
tmp_dir=.ds_tmp

dx=256
dy=256
dz=256

show_help() {
	echo "Usage: $(basename $0) -hio"
	echo "  -h: show help"
	echo "  -i <path>: raw input file path"
	echo "  -o <path>: output directory"
	echo "  -b <int>: log(block_size)"
	echo "  -p <int>: padding"
	echo "  -l <int>: sample levels"
}

while getopts "h?i:o:b:p:l:" opt; do
	case "$opt" in
		h|\?)
			show_help
			exit 0
			;;
		i)  raw_path=$OPTARG
			;;
		o)  dst_dir=$OPTARG
			;;
		b)  log_bs=$OPTARG
			;;
		p)  pad=$OPTARG
			;;
		l)  lvls=$OPTARG
			;;
	esac
done

if [ -z "$raw_path" ]; then
	show_help
	exit 0
fi

if [ ! -f "$raw_path" ]; then
	echo "raw path '$raw_path' does not exist"
	exit 0
fi

shift $((OPTIND-1))
[ "${1:-}" = "--" ] && shift


out=($(perl infer_dimension.pl ${raw_path}))
if [ $? != 0 ]; then
	echo "failed to infer dimension from raw filepath: '$raw_path"
	exit 0
fi
dx=${out[0]}
dy=${out[1]}
dz=${out[2]}

raw_base=$(basename ${raw_path})

echo "raw=$dx,$dy,$dz"
echo "padding=$pad; block_size=$(($log_bs))"
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
        | tee ${tmp_dir}/archiver.log

    ${toolkit_dir}/archive-thumbnailer \
            -o ${dst_dir} \
            -i ${dst_dir}/${lvl_raw_base%.*}_$((1 << ${log_bs}))p${pad}.h264 \
            -x chebyshev \
        | tee ${tmp_dir}/archive-thumbnailer.log
}

# set -x

mb=$(($dx*$dy*$dz / 1024 / 1024))
echo "sampling level 0/${lvls}... [$dx, $dy, $dz] = $mb MB"
lvl_arch ${raw_path} $dx $dy $dz

for i in $(seq $((${lvls}-1)))
do
	xx=$(($dx >> ($i)))
	yy=$(($dy >> ($i)))
	zz=$(($dz >> ($i)))

	mb=$(($xx*$yy*$zz / 1024 / 1024))
    echo "sampling level $i/${lvls}... [$xx, $yy, $zz] = $mb MB"

    ${toolkit_dir}/downsampler -i ${raw_path} -x $dx -y $dy -z $dz -s $i -o ${tmp_dir}\
        | tee ${tmp_dir}/downsampler.log

    lvl_raw_path=${tmp_dir}/${raw_base%.*}_x$((1 << $i)).raw

    lvl_arch ${lvl_raw_path} $(($dx >> i)) $(($dy >> i)) $(($dz >> i))
done

echo "writing package meta..."

${toolkit_dir}/meta-builder \
        -d ${dst_dir} \
    | tee ${tmp_dir}/meta-builder.log

# set +x

echo "built ${dst_dir}"
