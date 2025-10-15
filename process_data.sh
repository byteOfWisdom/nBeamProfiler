rm -f /tmp/data_conv_ppl /tmp/data_prep_ppl /tmp/data_scint_ppl

# mkfifo /tmp/data_conv_ppl
# mkfifo /tmp/data_prep_ppl
# mkfifo /tmp/data_scint_ppl

source ~/.zshrc

python3 square_scint.py $1 2.54 30. > /tmp/data_scint_ppl 

if [ $3 = "conv" ]
then
  python3 convert_ff.py $2 /tmp/data_conv_ppl 
else
  cat $2 > /tmp/data_conv_ppl 
fi

python3 prep_data.py /tmp/data_conv_ppl /tmp/data_prep_ppl 

julia -i deconv.jl /tmp/data_prep_ppl /tmp/data_scint_ppl 5000

# cat /tmp/data_prep_ppl

rm /tmp/data_conv_ppl /tmp/data_prep_ppl /tmp/data_scint_ppl
