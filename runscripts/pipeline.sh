#!/bin/bash
echo "Running focus_pipe.sh"
echo "======================"
echo "Running unzipper.sh"
source runscripts/unzipper.sh
echo "======================"
echo "Running decoder.sh"
source runscripts/decoder.sh
echo "======================"
echo "Running focuser.sh"
source runscripts/focuser.sh
echo "======================"
