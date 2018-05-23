# coincrunch-terraform
TF Scripts to provision EC2 Instance to run computationally expensive tasks.


#### Purpose
To help spin up ds environment for dsworkflows. You can do it manually too, but this is
meant to speed things up.


#### Basics
Make sure you have Terraform installed:
```
# on a Mac
brew terraform
```

You should have a file as such for your AWS credentials:
```
ls -la ~/.aws/credentials
```


#### Usage
```bash
cd dsworkflows/terraform_script

# pulls dependencies
terraform get -update

# validate
terraform plan

# build the resources requested
terraform apply

# cleanup once if you don't need the resource anymore
terraform destroy
```
