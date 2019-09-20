#!/bin/sh
# Generate mock spectra at a range of pixel phases to search for systematics
# caused by pixel phase.

noise=""

for arg in "$@"
do
    if [ "$arg" == "--help" ] || [ "$arg" == "-h" ]
    then
        echo "Usage: -g or -i for Gaussian or integrated Gaussian mock data"
    fi
    if [ "$arg" == "-g" ]
    then
        type="g"
    fi
    if [ "$arg" == "-i" ]
    then
        type="i"
    fi
    if [ "$arg" == "-n" ] || [ "$arg" == "--noise" ]
    then
        noise="n"
    fi
done

echo "#Amplitude Mu Sigma Baseline"
#for i in `seq -0.5 0.001 0.501`
#do
echo    ./investigate_integrated_gaussian.py -amp -100000000 -mu 5500 -sigma 0.05 -base 180000000 -f ${i} -$type$noise
#    ./investigate_integrated_gaussian.py -amp -40000 -mu 5500 -sigma 0.05 -base 72000 -f ${i} -$type$noise
#done

