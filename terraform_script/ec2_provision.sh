#!/bin/bash
# ec2 instance specific provisions

echo 'ec2_provision.sh start...' 

#echo 'export IAMROLE=`curl -s http://169.254.169.254/latest/meta-data/iam/security-credentials/`' >> ~/.bashrc
#echo 'export AWS_ACCESS_KEY_ID=`curl -s http://169.254.169.254/latest/meta-data/iam/security-credentials/$IAMROLE | grep '"'"'\"AccessKeyId\" : *'"'"' | cut -f5 -d '\" \"' | cut -b2- | rev | cut -b3- | rev`' >> ~/.bashrc
#echo 'export AWS_SECRET_ACCESS_KEY=`curl -s http://169.254.169.254/latest/meta-data/iam/security-credentials/$IAMROLE | grep '"'"'\"SecretAccessKey\" : *'"'"' | cut -f5 -d '\" \"' | cut -b2- | rev | cut -b3- | rev`' >> ~/.bashrc
#echo 'export AWS_SESSION_TOKEN=`curl -s http://169.254.169.254/latest/meta-data/iam/security-credentials/$IAMROLE | grep '"'"'\"Token\" : *'"'"' | cut -f5 -d '\" \"' | rev | cut -b2- | rev`' >> ~/.bashrc

#setup root virtualenv
cd ~/dsworkflows
source setup

#xgboost
cd ~
git clone --recursive https://github.com/dmlc/xgboost.git
cd xgboost
./build.sh
cd python-package
python setup.py install

#lightgbm w/ gpu support
cd ~
git clone --recursive https://github.com/Microsoft/LightGBM 
cd LightGBM/python-package
python setup.py install --gpu

#get all data files from buckets
source ~/dsworkflows/terraform_script/s3bucket
aws s3 cp s3://$BUCKETNAME/train.csv ~/dsworkflows/data/train.csv
aws s3 cp s3://$BUCKETNAME/config.py ~/dsworkflows/config/config.py

echo 'source $HOME/dsworkflows/env/bin/activate' >> $HOME/.bashrc

echo 'ec2_provision.sh complete' 
