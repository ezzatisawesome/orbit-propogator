from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS module

import numpy as np
from datetime import datetime
from marshmallow import Schema, fields, ValidationError

from src.orbit import Orbit
from src.bodies import Earth
from src.dataclasses import ClassicalOrbitalElements


app = Flask(__name__)
CORS(app)

class InputSchema(Schema):
    from_state = fields.Bool()
    time = fields.Int(required=True)  # Time in milliseconds
    elements = fields.List(fields.Float(), required=True, validate=lambda x: len(x) == 6)
    propagation_span = fields.Int(required=True)
    propagation_step = fields.Int(required=True)


@app.route("/propagate", methods=["POST"])
def propagate():
    # Validate and deserialize input
    schema = InputSchema()
    try:
        body = schema.load(request.get_json())
    except ValidationError as err:
        return jsonify(err.messages), 400
    
    # Set up epoch
    time_ms = body['time']
    t0 = datetime.fromtimestamp(time_ms / 1000.0)

    # Pull out time variables
    span = body['propagation_span']
    signedDt = np.copysign(body['propagation_step'], span)

    # Set up orbit
    elements = body['elements']
    if body['from_state']:
        orbit = Orbit.from_state(np.array(elements), Earth, t0)
    else:
        coes = elements
        sma = coes[0]
        ecc = coes[1]
        inc = coes[2]
        raan = coes[3]
        aop = coes[4]
        ta = coes[5]
        coesSat = ClassicalOrbitalElements(sma, ecc, inc, raan, aop, ta)
        orbit = Orbit.from_coes(coesSat, Earth, t0)

    # Output:
    statesSat = []
    statesGeocSat = []

    # Propagate
    for i in range(abs(span)):
        states, statesGeoc = orbit.propagate(signedDt, 1)
        statesSat.append(states[0].tolist())
        statesGeocSat.append(statesGeoc[0].tolist())

    return jsonify({
        "message": "Ok",
        "statesSat": statesSat,
        "statesGeocSat": statesGeocSat
    }), 200


if __name__ == '__main__':  
   app.run() 