maxStates=11
maxStrLen=1000000001
maxK=10001

cnta=2
while [ $cnta -lt $maxStates ]
do
	cntb=100
	while [ $cntb -lt $maxK ]
	do
		cntc=1000000
		while [ $cntc -lt $maxStrLen ]
		do
			./dfa2 $cnta 10 $cntc $cntb >> timeanalysis
			cntc=$((cntc*10));
		done
		cntb=$((cntb*10));
	done
	
	cnta=$((cnta+1));
done