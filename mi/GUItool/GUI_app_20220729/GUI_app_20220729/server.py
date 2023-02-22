import os
from tool_app import app
import ssl
from werkzeug.serving import WSGIRequestHandler

if __name__ == "__main__":
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    ssl_folder = "/home/test_user/Documents/config"
    crt_filename = os.path.join(ssl_folder, "server.crt")
    key_filename = os.path.join(ssl_folder, "server.key")
    ssl_context.load_cert_chain(crt_filename, key_filename)
    WSGIRequestHandler.protocol_version = "HTTP/1.1" # HTTP/1.1対応の宣言

    app.run(host="0.0.0.0", debug=True, port=8000, ssl_context=ssl_context)