
#!/usr/bin/env bash
http_code=$(curl -m 10 -LI https://mrtg.synet.edu.cn -k -o /dev/null -w '%{http_code}
' -s)
if [ ${http_code} -eq 200 ]; then
echo "the ip is already online "
exit 0
fi
curl "https://ipgw.neu.edu.cn/ipgw-key/ippass?key=LtT_iDg5SpaeO152z0zxAjTui3eiekM7"
echo ""

