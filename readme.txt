To do a basic reduction
1. Edit fn_call.sh to have the correct settings and directories
2. "bash fn_call.sh"


To do reduction overnight
1. Edit fn_call.sh to have correct settings and directories
2. "nohup bash fn_call.sh &"


To open a jupyter notebook
1. ON astro-node "jupyter notebook --no-browser"
2. ON local computer "ssh -N -L localhost:8000:localhost:8888 astro-node11"
3. Then open browser to localhost:8000
4. (sometimes) copy/paste the token given in the astro-node terminal
