let net;

const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();

let str = localStorage.getItem("classifier");
if (str) {
  classifier.setClassifierDataset(Object.fromEntries(JSON.parse(str).map(([label, data, shape]) => [label, tf.tensor(data, shape)])));
}

async function app() {
  console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load();

  console.log('Successfully loaded model');

  // Create an object from Tensorflow.js data API which could capture image 
  // from the web camera as Tensor.
  const webcam = await tf.data.webcam(webcamElement);

  // Reads an image from the webcam and associates it with a specific class
  // index.
  const addExample = async classId => {
    // Capture an image from the web camera.
    const img = await webcam.capture();

    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    const activation = net.infer(img, true);

    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);

    // Dispose the tensor to release the memory.
    img.dispose();
    let strClassifier = JSON.stringify(Object.entries(classifier.getClassifierDataset()).map(([label, data]) => [label, Array.from(data.dataSync()), data.shape]));
    localStorage.setItem("classifier", strClassifier);
  };

  // When clicking a button, add an example for that class.
  document.getElementById('class-a').addEventListener('click', () => addExample(0));
  document.getElementById('class-b').addEventListener('click', () => addExample(1));
  document.getElementById('class-c').addEventListener('click', () => addExample(2));
  document.getElementById('class-d').addEventListener('click', () => addExample(3));
  document.getElementById('class-e').addEventListener('click', () => addExample(4));
  document.getElementById('class-f').addEventListener('click', () => addExample(5));
  document.getElementById('class-g').addEventListener('click', () => addExample(6));
  document.getElementById('class-h').addEventListener('click', () => addExample(7));
  document.getElementById('class-i').addEventListener('click', () => addExample(8));
  document.getElementById('class-j').addEventListener('click', () => addExample(9));
  document.getElementById('class-k').addEventListener('click', () => addExample(10));
  document.getElementById('class-l').addEventListener('click', () => addExample(11));
  document.getElementById('class-m').addEventListener('click', () => addExample(12));
  document.getElementById('class-n').addEventListener('click', () => addExample(13));
  document.getElementById('class-o').addEventListener('click', () => addExample(14));

  document.getElementById('download').addEventListener('click', download);

  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture();

      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(img, 'conv_preds');
      // Get the most likely class and confidence from the classifier module.
      const result = await classifier.predictClass(activation);

      const classes = [
        'Zero,0',
        'Um,1',
        'Dois,2',
        'Três,3',
        "Quatro,4",
        "Cinco,5",
        "Seis,6",
        "Sete,7",
        "Oito,8",
        "Nove,9",
        "Bloqueio,20",
        "Seta,20",
        "Mais Dois,20",
        "Coringa,50",
        "Nenhuma Carta,0"
      ];
      let splitClass = classes[result.label].split(",");
      document.getElementById('console').innerText = `
        Previsão: ${splitClass[0]}\n
        Pontos: ${splitClass[1]}\n
        Probabilidade: ${result.confidences[result.label] * 100 + "%"}
      `;

      // Dispose the tensor to release the memory.
      img.dispose();
    }

    await tf.nextFrame();
  }
}

app();

function download() {
  let strModel = localStorage.getItem("classifier");
  if(!strModel){
    return;
  }

  var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(strModel);
  
  var dlAnchorElem = document.createElement('a');
  dlAnchorElem.setAttribute("href", dataStr);
  dlAnchorElem.setAttribute("download", "model.json");
  dlAnchorElem.click();
}
document.getElementById('uploadFile').addEventListener('change', onChange);

function onChange(event) {
  var reader = new FileReader();
  reader.onload = onReaderLoad;
  reader.readAsText(event.target.files[0]);
}

function onReaderLoad(event){
  var obj = JSON.parse(event.target.result);
  if(obj){
    localStorage.setItem("classifier",JSON.stringify(obj));
    window.location.reload();
  }
}