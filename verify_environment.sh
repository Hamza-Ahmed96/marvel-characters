#!/bin/bash
echo "=== Environment Verification ==="
echo ""
echo "1. Virtual environment location:"
ls -ld .venv
echo ""
echo "2. Python interpreter:"
.venv/bin/python --version
echo ""
echo "3. Python path:"
echo "$PWD/.venv/bin/python"
echo ""
echo "4. Installed packages (Marvel project):"
.venv/bin/pip list | grep -E "marvel|databricks|mlflow"
echo ""
echo "5. Marvel-teach command:"
ls -lh .venv/bin/marvel-teach
echo ""
echo "=== ✅ If all above show results, your environment is ready! ==="
