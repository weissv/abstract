#!/bin/bash
echo "🔍 Verifying Project Refactoring..."
echo ""

# Check for hardcoded tokens
echo "1️⃣ Checking for hardcoded tokens..."
if grep -r "hf_dxm" --include="*.py" --include="*.yaml" . 2>/dev/null | grep -v README_OLD; then
    echo "   ❌ Found hardcoded tokens!"
else
    echo "   ✅ No hardcoded tokens found"
fi

# Check model name
echo ""
echo "2️⃣ Checking model name..."
if grep -q "Meta-Llama-3.1-8B-Instruct" config.yaml; then
    echo "   ✅ Using Llama-3.1"
else
    echo "   ❌ Wrong model version"
fi

# Check device config
echo ""
echo "3️⃣ Checking device configuration..."
if grep -q 'device: "cuda"' config.yaml; then
    echo "   ✅ CUDA device configured"
else
    echo "   ❌ Wrong device configuration"
fi

# Check 4-bit quantization
echo ""
echo "4️⃣ Checking quantization..."
if grep -q "load_in_4bit: true" config.yaml; then
    echo "   ✅ 4-bit quantization enabled"
else
    echo "   ❌ Quantization not enabled"
fi

# Check required files
echo ""
echo "5️⃣ Checking required files..."
files=("README.md" "QUICKSTART.md" "CONTRIBUTING.md" "LICENSE" "requirements.txt" "setup.py" ".gitignore" "llama_refusal_analysis.ipynb")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file exists"
    else
        echo "   ❌ $file missing"
    fi
done

# Check experiment scripts
echo ""
echo "6️⃣ Checking experiment scripts..."
for exp in experiments/01_baseline.py experiments/02_patching.py experiments/03_ablation.py; do
    if grep -q "hf_token=None" "$exp"; then
        echo "   ✅ $(basename $exp) - token removed"
    else
        echo "   ❌ $(basename $exp) - still has hardcoded token"
    fi
done

echo ""
echo "🎉 Verification complete!"
