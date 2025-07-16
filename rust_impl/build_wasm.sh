#!/bin/bash

# Build script for WASM deployment

echo "Building WASM module for prime factorization..."

# Install wasm-pack if not already installed
if ! command -v wasm-pack &> /dev/null; then
    echo "Installing wasm-pack..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Clean previous builds
rm -rf pkg/

# Build the WASM module
echo "Building WASM module..."
wasm-pack build --target web --features wasm --no-default-features

# Create example HTML file
cat > pkg/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Prime Factorization WASM Demo</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            margin: 20px 0;
        }
        input[type="text"] {
            width: 300px;
            padding: 5px;
            font-size: 16px;
        }
        button {
            padding: 5px 15px;
            font-size: 16px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        #result {
            margin-top: 20px;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 5px;
            min-height: 50px;
        }
        .error {
            color: red;
        }
        .success {
            color: green;
        }
    </style>
</head>
<body>
    <h1>Prime Factorization WASM Demo</h1>
    
    <div class="container">
        <h2>Single Number Factorization</h2>
        <input type="text" id="numberInput" placeholder="Enter a number to factorize" value="2539123152460219">
        <button onclick="factorize()">Factorize</button>
        <div id="result"></div>
    </div>
    
    <div class="container">
        <h2>Batch Factorization</h2>
        <textarea id="batchInput" rows="4" cols="50" placeholder="Enter numbers (one per line)">100822548703
123456789
97
1000000007</textarea>
        <br>
        <button onclick="factorizeBatch()">Factorize Batch</button>
        <div id="batchResult"></div>
    </div>

    <script type="module">
        import init, { WasmFactorizer } from './genomic_pleiotropy_cryptanalysis.js';
        
        let factorizer;
        
        async function initialize() {
            await init();
            factorizer = new WasmFactorizer();
            console.log("WASM module initialized");
        }
        
        window.factorize = async function() {
            const number = document.getElementById('numberInput').value;
            const resultDiv = document.getElementById('result');
            
            try {
                resultDiv.innerHTML = '<span>Factorizing...</span>';
                const result = await factorizer.factorize_async(number);
                const data = JSON.parse(result);
                
                resultDiv.innerHTML = `
                    <div class="success">
                        <strong>Success!</strong><br>
                        Number: ${data.number}<br>
                        Factors: ${data.factors.join(' × ')}<br>
                        Time: ${data.elapsed_ms.toFixed(2)}ms<br>
                        Used CUDA: ${data.used_cuda}
                    </div>
                `;
            } catch (error) {
                resultDiv.innerHTML = `<div class="error">Error: ${error}</div>`;
            }
        };
        
        window.factorizeBatch = async function() {
            const input = document.getElementById('batchInput').value;
            const numbers = input.split('\n').filter(n => n.trim());
            const resultDiv = document.getElementById('batchResult');
            
            try {
                resultDiv.innerHTML = '<span>Factorizing batch...</span>';
                const result = await factorizer.factorize_batch(JSON.stringify(numbers));
                const data = JSON.parse(result);
                
                let html = '<div class="success"><strong>Batch Results:</strong><br>';
                for (const item of data) {
                    html += `${item.number} = ${item.factors.join(' × ')} (${item.elapsed_ms.toFixed(2)}ms)<br>`;
                }
                html += '</div>';
                
                resultDiv.innerHTML = html;
            } catch (error) {
                resultDiv.innerHTML = `<div class="error">Error: ${error}</div>`;
            }
        };
        
        // Initialize on load
        initialize();
    </script>
</body>
</html>
EOF

echo "WASM build complete! Files generated in pkg/"
echo "To test locally, run: python3 -m http.server 8000 --directory pkg/"
echo "Then open http://localhost:8000 in your browser"