cnt=100000000
while [ $cnt -lt 1000000000 ]
do
	./naive gen_file out $cnt $cnt 0
	cnt=$((cnt+100000000));
done