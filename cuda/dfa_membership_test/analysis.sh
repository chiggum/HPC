maxStates=6
maxStrLen=5000000001
maxK=10001

cnta=2
while [ $cnta -lt $maxStates ]
do
	cntb=100
	while [ $cntb -lt $maxK ]
	do
		cntc=100000
		while [ $cntc -lt $maxStrLen ]
		do
			./dfa2 $cnta 10 $cntc $cntb >> timeanalysis_cuda.dat
			cntc=$((cntc*2));
		done
		cntb=$((cntb*10));
	done
	
	cnta=$((cnta+1));
done