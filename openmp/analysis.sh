cnt=2
while [ $cnt -lt 10000000 ]
do
	./naive gen_file out $cnt $cnt 0
	cnt=$((cnt*2));
done