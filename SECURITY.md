# Security Policy

## Reporting Security Issues

If you discover a security vulnerability, please email abylaidospayev@gmail.com instead of using the issue tracker.

## Sensitive Information

**NEVER commit the following to this repository:**

- Passwords or API keys
- Trading account credentials
- `.env` files with real credentials
- RDP connection files (`.rdp`)
- VM credentials or access tokens
- Private keys or certificates
- Database connection strings

## Safe Practices

### For Contributors

1. **Always use `.env.example`**: Never commit actual `.env` files
2. **Check before committing**: Review changes with `git diff` before commit
3. **Use .gitignore**: Ensure sensitive files are in `.gitignore`
4. **Local testing only**: Keep credentials in local files only

### For Users

1. **Create your own `.env`**: Copy from `.env.example` and fill with your credentials
2. **Never share credentials**: Don't share your `.env` or credential files
3. **Use demo accounts first**: Always test with demo accounts before live trading
4. **Rotate credentials**: Change passwords regularly, especially after sharing code

### Files That Should NEVER Be Committed

- `live_trading/.env`
- `*credentials*.txt`
- `*password*.txt`
- `*.rdp`
- Any file containing real API keys or passwords

## What To Do If Credentials Are Exposed

If you accidentally commit credentials:

1. **Immediately rotate/change** all exposed credentials
2. **Remove from git history** using `git filter-branch` or `git filter-repo`
3. **Force push** to update remote repository
4. **Notify affected parties** if credentials were shared

## Secure Configuration

The repository includes:
- `.gitignore` configured to exclude sensitive files
- `.env.example` as a template (no real credentials)
- Documentation on secure setup

## Trading Security

- **Never trade with money you can't afford to lose**
- **Use demo accounts** for testing
- **Enable 2FA** on your broker account
- **Monitor positions** regularly
- **Set stop losses** on all positions
- **Limit daily loss** with circuit breakers

## Code Security

- All Python packages are listed in `requirements.txt`
- Keep dependencies up to date
- Review third-party code before use
- Don't run code from untrusted sources

## Contact

For security concerns: abylaidospayev@gmail.com
