#!/usr/bin/env bash
# Clear corrupted Cartopy cache files

echo "Clearing Cartopy cache..."

# Cartopy cache is typically in ~/.local/share/cartopy
CACHE_DIR="$HOME/.local/share/cartopy"

if [ -d "$CACHE_DIR" ]; then
    echo "Found Cartopy cache at: $CACHE_DIR"
    du -sh "$CACHE_DIR"
    
    read -p "Delete this cache? (yes/no): " CONFIRM
    
    if [ "$CONFIRM" = "yes" ]; then
        rm -rf "$CACHE_DIR"
        echo "✓ Cache cleared"
        echo "Cartopy will re-download shapefiles on next run"
    else
        echo "Cancelled"
    fi
else
    echo "No Cartopy cache found at $CACHE_DIR"
fi

# Also check alternative locations
ALT_CACHE="$HOME/.cache/cartopy"
if [ -d "$ALT_CACHE" ]; then
    echo ""
    echo "Found alternative cache at: $ALT_CACHE"
    du -sh "$ALT_CACHE"
    
    read -p "Delete this cache too? (yes/no): " CONFIRM2
    
    if [ "$CONFIRM2" = "yes" ]; then
        rm -rf "$ALT_CACHE"
        echo "✓ Alternative cache cleared"
    fi
fi

echo ""
echo "Done! Next plot job will download fresh shapefiles."
