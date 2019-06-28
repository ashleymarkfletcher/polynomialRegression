// store mouse click points
const x_vals = []
const y_vals = []

let a
let b
let c
let d

const learningRate = 0.1

// optimzer works on ALL tf variables if no sencond parameter is put in
const optimizer = tf.train.adam(learningRate)

function setup() {
  createCanvas(400, 400)
  background(0)

  // init m and b as random values between -1 and 1
  // they are tf vriables because they change over time
  a = tf.variable(tf.scalar(random(-1, 1)))
  b = tf.variable(tf.scalar(random(-1, 1)))
  c = tf.variable(tf.scalar(random(-1, 1)))
  d = tf.variable(tf.scalar(random(-1, 1)))
}

const loss = (predictions, labels) =>
  predictions
    .sub(labels)
    .square()
    .mean()

const predict = x => {
  const xs = tf.tensor1d(x)

  // formula for a polynomial/curve
  // y = ax^2 + bc + c
  // const ys = xs
  //   .square()
  //   .mul(a)
  //   .add(xs.mul(b))
  //   .add(c)

  // cubed
  const ys = xs
    .pow(tf.scalar(3))
    .mul(a)
    .add(xs.square().mul(b))
    .add(xs.mul(c))
    .add(d)

  return ys
}

function mousePressed() {
  // normalise the graph from pixels
  const x = map(mouseX, 0, width, -1, 1)
  const y = map(mouseY, 0, width, 1, -1)

  x_vals.push(x)
  y_vals.push(y)
}

function draw() {
  background(0)
  stroke(255)
  strokeWeight(8)

  // get rid of all unneaded tensors to stop memory leak
  tf.tidy(() => {
    // don't train if no points
    if (x_vals.length > 0) {
      const ys = tf.tensor1d(y_vals)
      optimizer.minimize(() => loss(predict(x_vals), ys))
    }
  })

  // draw the points
  for (let i = 0; i < x_vals.length; i++) {
    // undo the normalising
    const px = map(x_vals[i], -1, 1, 0, width)
    const py = map(y_vals[i], -1, 1, height, 0)

    point(px, py)
  }

  // points at the left and right 0-1
  const curveX = []

  for (let x = -1; x < 1.01; x += 0.05) {
    curveX.push(x)
  }

  // predict the Y for each x point
  const ys = tf.tidy(() => predict(curveX))
  // get the values in the prediction
  const curveY = ys.dataSync()
  // get rid of ys since it's not needed
  ys.dispose()

  // start drawing the curve
  beginShape()
  noFill()
  strokeWeight(2)

  for (let i = 0; i < curveX.length; i++) {
    // map from normalised back to pixels
    // first point
    const x = map(curveX[i], -1, 1, 0, width)
    const y = map(curveY[i], -1, 1, height, 0)

    vertex(x, y)
  }

  endShape()
}
