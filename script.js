let model;
let modelLoaded = false; // Variabel untuk mengecek status pemuatan model

console.log("TensorFlow.js version:", tf.version.tfjs);

// Fungsi untuk memuat model dari file model.json
async function loadModel() {
    console.log("Loading model...");
    try {
        model = await tf.loadGraphModel('./tfjs_model/model.json');
        modelLoaded = true; // Tandai model sudah dimuat
        console.log("Model loaded.");
        document.getElementById('classifyButton').disabled = false; // Aktifkan tombol classify setelah model dimuat
    } catch (error) {
        console.error("Error loading model:", error);
        alert("Failed to load the model. Please check the path or try again.");
    }
}

// Fungsi untuk menangani input gambar
document.getElementById('imageInput').addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = () => {
            const img = document.getElementById('uploadedImage');
            img.src = reader.result;
            img.style.display = 'block'; // Menampilkan gambar yang diunggah
        };
        reader.readAsDataURL(file);
    }
});

// Fungsi untuk memproses gambar dan memprediksi
async function classifyImage() {
    if (!modelLoaded) {
        alert("Model is not loaded yet. Please wait until the model is loaded.");
        return;
    }

    const image = document.getElementById('uploadedImage');
    if (!image.src) {
        alert("Please upload an image first.");
        return;
    }

    // Proses gambar sebelum melakukan prediksi
    const tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([177, 177]) // Ukuran yang sesuai dengan model
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims(0); // Menambahkan batch dimension

    try {
        const rawPredictions = await model.predict(tensor).data(); // Prediksi mentah dari model
        const normalizedPredictions = normalizePredictions(rawPredictions); // Normalisasi ke skala 0-1
        console.log("Normalized Predictions: ", normalizedPredictions);

        const percentagePredictions = normalizedPredictions.map((value) => value * 100); // Konversi ke persentase
        console.log("Percentage Predictions: ", percentagePredictions);

        const classNames = ['Anggur', 'Apel', 'Belimbing', 'Jeruk', 'Kiwi', 'Mangga', 'Nanas', 'Pisang', 'Semangka', 'Stroberi'];

        // Urutkan prediksi berdasarkan nilai tertinggi ke terendah
        const sortedIndices = Array.from(percentagePredictions.keys())
            .sort((a, b) => percentagePredictions[b] - percentagePredictions[a]);

        // Update taskbars berdasarkan prediksi
        updateTaskbars(percentagePredictions, classNames, sortedIndices);
    } catch (error) {
        console.error("Error during prediction:", error);
        alert("Prediction failed. Please try again.");
    }
}

// Fungsi untuk melakukan normalisasi prediksi menggunakan softmax
function normalizePredictions(predictions) {
    const expPredictions = predictions.map((value) => Math.exp(value));
    const sumExpPredictions = expPredictions.reduce((a, b) => a + b, 0);
    return expPredictions.map((value) => value / sumExpPredictions); // Normalisasi ke rentang 0-1
}

// Fungsi untuk mengupdate taskbars berdasarkan prediksi
function updateTaskbars(predictions, classNames, sortedIndices) {
    const taskbarsContainer = document.getElementById('taskbarsContainer');
    taskbarsContainer.innerHTML = ''; // Kosongkan kontainer sebelum diisi ulang

    // Perbarui taskbars berdasarkan urutan tertinggi ke terendah
    sortedIndices.forEach((index) => {
        const percentage = predictions[index].toFixed(2); // Ambil nilai prediksi
        const className = classNames[index]; // Ambil nama kelas

        // Buat elemen taskbar
        const taskbar = document.createElement('div');
        taskbar.className = 'taskbar';
        taskbar.id = `taskbar${className}`;

        // Buat elemen label
        const label = document.createElement('span');
        label.textContent = `${className}: ${percentage}%`;

        // Buat elemen progress bar
        const progressBar = document.createElement('div');
        progressBar.className = 'progress-bar';
        progressBar.style.width = `${percentage}%`;
        progressBar.style.backgroundColor = `rgba(76, 175, 80, ${predictions[index] / 100})`; // Warna hijau dengan opacity sesuai

        // Gabungkan ke dalam taskbar
        taskbar.appendChild(label);
        taskbar.appendChild(progressBar);

        // Tambahkan taskbar ke kontainer
        taskbarsContainer.appendChild(taskbar);
    });
}

// Memuat model saat halaman dimuat
window.onload = () => {
    loadModel();

    // Menambahkan event listener ke tombol classify
    const classifyButton = document.getElementById('classifyButton');
    classifyButton.addEventListener('click', classifyImage);

    // Nonaktifkan tombol "Classify" saat model belum dimuat
    classifyButton.disabled = true;
};
