#!/bin/bash
#w-ade
/usr/bin/yum -y install httpd php php-mysqli
/sbin/chkconfig httpd on
/sbin/service httpd start
