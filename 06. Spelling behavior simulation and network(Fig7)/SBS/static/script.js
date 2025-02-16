document.getElementById('paramsForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const params = {};
    formData.forEach((value, key) => { params[key] = parseFloat(value); });

    fetch('/run_simulation', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(params)
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('resultImage').src = 'data:image/png;base64,' + data.image;
    })
    .catch(error => console.error('Error:', error));
});
