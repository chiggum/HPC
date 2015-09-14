cnt=20
while [ $cnt -lt 1005 ]
do
	./a.out 5 5 $cnt >> output
	cnt=$((cnt+1));
done