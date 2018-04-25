from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers.pyspark import deploy_pyspark_model
from clipper_admin.deployers import python as python_deployer

clipper_conn = ClipperConnection(DockerContainerManager())
clipper_conn.start_clipper()
clipper_addr = clipper_conn.get_query_addr()

def preprocess(inputs):
	inputArr = (inputs[0]).split(",")
	floats = inputArr[:-1]
	rounded = [round(float(i),1) for i in floats]
	rounded.append(inputArr[-1])
	output = [(str(rounded))[1:-1]]
	return output

python_deployer.deploy_python_closure(
    clipper_conn,
    name="process-iris",  # The name of the model in Clipper
    version=1,  # A unique identifier to assign to this model.
    input_type="string",  # The type of data the model function expects as input
    func=preprocess # The model function to deploy
)

clipper_conn.register_application(
    name="process-app",
    input_type="strings",
    default_output="-1",
    slo_micros=9000000) #will return default value in 9 seconds

clipper_conn.link_model_to_app(app_name="process-app", model_name="process-iris")


