#!/bin/sh
# This works for cross region
outputFile="requirements.txt"
awsFile="cloud-stack-overflow/requirements.txt"
bucket="cloud-stack-overflow"
resource="/${bucket}/${awsFile}"
contentType="application/x-compressed-tar"

# Change the content type as desired
dateValue=`TZ=GMT date -R`
#Use dateValue=`date -R` if your TZ is already GMT
# stringToSign="GET\n\n${contentType}\n${dateValue}\n${resource}"
# s3Key="ACCESS_KEY_ID"
# s3Secret="SECRET_ACCESS_KEY"
# signature=`echo -n ${stringToSign} | openssl sha1 -hmac ${s3Secret} -binary | base64`
curl -H "Host: ${bucket}.s3.amazonaws.com" \
     -H "Content-Type: ${contentType}" \
     https://${bucket}.s3.amazonaws.com/${awsFile} -o $outputFile

# Install the pip requirements
pip install -r requirements.txt
