import uwsgi
from uwsgidecorators import filemon

filemon("/home/matsu_user1/Documents/app/tool_app")(uwsgi.reload)
filemon("/home/matsu_user1/Documents/app/tybmilib")(uwsgi.reload)