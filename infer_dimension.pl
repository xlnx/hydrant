#!/usr/bin/perl

$fname = $ARGV[0];

my ( $dx, $dy, $dz ) = ( $fname =~ m/(\d+)x(\d+)x(\d+)/ );

if ( !defined $dx ) {
	exit 1;
} else {
	print "$dx $dy $dz\n";
}
