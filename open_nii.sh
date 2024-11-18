#!/bin/sh
if [ -d "export" ];then
    if [ -z "$( ls -A 'export' )" ]; then
        echo "export folder is Empty"
    else
        echo 'export folder is Not Empty'
        gzip export/*
	    tar czvf export_prior.tar.gz ./export/*.gz
	    rm export/*
    fi

	rm -d export
fi

if [ -d "images" ];then
    if [ -z "$( ls -A 'images' )" ]; then
        echo "images Empty"
    else
        echo 'images is Not Empty'
        gzip images/*
	    tar czvf images_prior.tar.gz ./images/*.gz
	    rm images/*
    fi
	rm -d images
fi

mkdir images
mkdir export

if [ -d "compressed_archives" ];then
    if [ -z "$( ls -A 'compressed_archives' )" ]; then
        echo "Compressed Archives is Empty"
    else
        echo 'Compressed Archives is Not Empty'
        gunzip compressed_archives/*.gz
        cp compressed_archives/* images
        echo 'Executing open.py'
        python open.py
        gzip compressed_archives/*
    fi
fi
