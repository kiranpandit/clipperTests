package tinytest

import scala.concurrent.duration._

import io.gatling.core.Predef._
import io.gatling.http.Predef._
import io.gatling.jdbc.Predef._

class ClipperSimulation extends Simulation {

	//val ints_small = csv("gauss.csv").random
	//val ints_small = csv("uniform.csv").random
	val ints_small = csv("adultLabeled.csv").random

	val httpProtocol = http
		.baseURL("http://192.168.0.105:1337")
		.inferHtmlResources()
		.acceptHeader("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
		.acceptEncodingHeader("gzip, deflate")
		.acceptLanguageHeader("en-US,en;q=0.5")
		.userAgentHeader("Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:58.0) Gecko/20100101 Firefox/58.0")


object Load {
	val load = feed(ints_small)
			  .exec(http("request_0")
			  .post("/pyspark-app/predict")
			  .body(StringBody("""{"input": [${x1}.0,${x2}.0,${x3}.0,${x4}.0,${x5}.0,${x6}.0,${x7}.0,${x8}.0,${x9}.0,${x10}.0,${x11}.0,${x12}.0,${x13}.0,${x14}.0,${x15}.0]}""")).asJSON
			  .header("Content-Type", "application/json"))
			.pause(1)
}

	val scn = scenario("ClipperSimulation").exec(Load.load)

	setUp(
		scn.inject(rampUsersPerSec(10) to 20 during (1 minutes),
		constantUsersPerSec(20) during (30 seconds)
		)
	).protocols(httpProtocol)
}

