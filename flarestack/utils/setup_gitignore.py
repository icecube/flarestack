from __future__ import print_function
import os
from flarestack.shared import fs_dir

print(fs_dir)

gitignore_path = fs_dir + ".git/info/exclude"

print(gitignore_path)

text = "# git ls-files --others --exclude-from=.git/info/exclude \n" \
       "# Lines that start with '#' are comments. \n" \
       "# For a project mostly in C, the following would be a good set of \n " \
       "# exclude patterns (uncomment them if you want to use them): \n" \
       "# *.[oa] \n" \
       "# *~ \n" \
       ".* \n" \
       "**/.* \n" \
       "**/**/.* \n" \
       "**/**/**/.* \n" \
       "**/**/**/**/.* \n" \
       "**.pyc \n" \
       "**.xml \n" \
       "**/config.py \n" \
       "build* \n" \
       "dist* \n" \
       "flarestack/cluster/SubmitDESY.sh \n " \
       "flarestack.egg-info/*"