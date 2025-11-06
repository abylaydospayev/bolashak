# OANDA API Token Setup Guide

## ⚠️ IMPORTANT: You Need an API Token (Not Password)

The credentials shown in the screenshot are:
- **Login:** 1600037272  
- **Master Password:** 8?p!?$$*kW3 (for web/platform login)
- **Read-only Password:** DXf22y?*@! (for view-only access)

**These are NOT API tokens!** You need to generate a Personal Access Token (PAT).

---

## Steps to Get Your OANDA API Token

### 1. Log into OANDA

Go to: https://www.oanda.com/demo-account/login

Use:
- **Username:** 1600037272
- **Password:** 8?p!?$$*kW3

### 2. Navigate to API Settings

Once logged in:
1. Click on your account name (top right)
2. Go to **"Manage API Access"** or **"API Settings"**
3. Or direct link: https://www.oanda.com/demo-account/tpa/personal_token

### 3. Generate Personal Access Token (PAT)

1. Click **"Generate Token"** or **"Create Token"**
2. Copy the token (it looks like: `a1b2c3d4e5f6...` - long string)
3. **SAVE IT IMMEDIATELY** - you can't see it again!

### 4. Update Configuration

Once you have the token, update `live_trading/oanda_config.py`:

```python
OANDA_CONFIG = {
    'account_id': '1600037272',
    'api_token': 'YOUR_API_TOKEN_HERE',  # Paste your PAT here
    'environment': 'practice',
    'server': 'OANDA-Demo-1',
}
```

---

## Alternative: Use Environment Variable (More Secure)

Instead of hardcoding the token, use environment variable:

### Windows PowerShell:
```powershell
$env:OANDA_API_TOKEN = "your_token_here"
```

### Then update code:
```python
import os

OANDA_CONFIG = {
    'account_id': '1600037272',
    'api_token': os.getenv('OANDA_API_TOKEN'),
    'environment': 'practice',
}
```

---

## Next Steps

1. ✅ Get your API token from OANDA website
2. ✅ Update `oanda_config.py` with the token
3. ✅ Run test: `python live_trading/oanda_client.py`
4. ✅ If successful → Deploy trading bot

---

## Troubleshooting

### "401 Unauthorized"
- Wrong API token → Regenerate token

### "400 Bad Request"
- Wrong account ID → Check account number
- Wrong environment → Use 'practice' for demo

### "403 Forbidden"
- Token doesn't have permissions → Generate new token with full access

---

## Security Notes

⚠️ **NEVER** commit API tokens to Git!
⚠️ **NEVER** share API tokens publicly!
✅ **DO** use environment variables
✅ **DO** regenerate tokens if compromised

---

Once you have the API token, paste it here and I'll update the config file for you! 🔑
