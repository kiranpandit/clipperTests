# A little script to stop clipper connections (for when you forget/it returns an error)
from clipper_admin import ClipperConnection, DockerContainerManager
clipper_conn = ClipperConnection(DockerContainerManager())
clipper_conn.connect()
clipper_conn.stop_all()