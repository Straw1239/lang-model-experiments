import sys


exec('from ' + sys.argv[1] + ' import *')
exec(sys.argv[2])
