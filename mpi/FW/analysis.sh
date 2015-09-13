sz=( 10, 50, 100, 500, 1000, 2000, 5000)
np=( 1, 2, 4, 8 )
for i in "${sz[@]}"
do
	:
	printf "Generating %sX%s matrix\n" "$i" "$i"
	./$1 $i gen_out

	printf "Executing Serial FW on %sX%s matrix\n" "$i" "$i"
	./$2 $i gen_out seq_out

	for j in "${np[@]}"
	do
		:
		printf "Executing mpi FW with %s proc on %sX%s matrix\n" "$j" "$i" "$i"
		mpiexec -n $j ./$3 $i gen_out mpi_out
		x=$( diff seq_out mpi_out )
		if [ "$x" = "" ]; then
			printf "Correct output.\n"
		else
			printf "Incorrect output\n"
		fi
		printf "\n"
	done
	printf "\n"
done