from flask import Flask, request, jsonify
from datetime import datetime
from marshmallow import Schema, fields, ValidationError

from src.orbit import Orbit
from src.bodies import Earth
from src.dataclasses import ClassicalOrbitalElements


app = Flask(__name__)

class InputSchema(Schema):
    time = fields.Int(required=True)  # Time in milliseconds
    coes = fields.List(fields.Float(), required=True, validate=lambda x: len(x) == 6)
    propagation_span = fields.Int(required=True)
    propagation_step = fields.Int(required=True)


@app.route("/", methods=["POST"])
def hello_world():
    # Validate and deserialize input
    schema = InputSchema()
    try:
        data = schema.load(request.get_json())
    except ValidationError as err:
        return jsonify(err.messages), 400

    coes = data['coes']
    span = data['propagation_span']
    dt = data['propagation_step']

    # Epoch
    time_ms = data['time']
    t0 = datetime.fromtimestamp(time_ms / 1000.0)

    # Satellite orbit
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

    # Propagate 1 minute
    for i in range(span):
        states, statesGeoc = orbit.propagate(dt, 1)
        statesSat.append(states[0].tolist())
        statesGeocSat.append(statesGeoc[0].tolist())

    return jsonify({
        "message": "Ok",
        "statesSat": statesSat,
        "statesGeocSat": statesGeocSat
    }), 200


if __name__ == '__main__':  
   app.run()  