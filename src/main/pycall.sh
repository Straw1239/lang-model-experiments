tmpfile=$(mktemp pycall.XXXXXX)
exec 3<>"$tmpfile"
rm "$tmpfile"

cat $1 <(echo $2) >&3
cat <&3
python3.7 <&3
