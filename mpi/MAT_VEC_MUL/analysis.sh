sz=( 10 50 100 500 1000 2000 5000 10000 20000 )
szz=( 5 25 50 250 500 1000 2500 5000 10000 )
np=( 4 )
cnt="0"
while [ "$cnt" != "9" ]
do
	:
	i=${sz[cnt]}
	ip=${szz[cnt]}
	printf "Generating %sX%s matrix\n" "$i" "$i"
	./$1 $i $i gen_out

	printf "Executing Serial code on %sX%s matrix\n" "$i" "$i"
	./$2 $i $i $ip $ip gen_out seq_out

	for j in "${np[@]}"
	do
		:
		printf "Executing mpi code with %s proc on %sX%s matrix\n" "$j" "$i" "$i"
		mpiexec -n $j ./$3 $i $i $ip $ip gen_out mpi_out
		x=$( diff seq_out mpi_out )
		if [ "$x" = "" ]; then
			printf "Correct output.\n"
		else
			printf "Incorrect output\n"
		fi
		printf "\n"
	done
	printf "\n"
	cnt=$[$cnt+1]
done