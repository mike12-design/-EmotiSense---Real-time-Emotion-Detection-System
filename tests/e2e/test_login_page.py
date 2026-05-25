import pytest


pytestmark = pytest.mark.e2e


def test_login_page_renders(page, frontend_base_url):
    try:
        page.goto(f"{frontend_base_url}/login", wait_until="domcontentloaded", timeout=10000)
    except Exception as exc:
        pytest.skip(f"Frontend server is not reachable at {frontend_base_url}: {exc}")

    expect_title = page.get_by_role("heading", name="EmotiSense")
    expect_username = page.get_by_placeholder("请输入用户名")
    expect_password = page.get_by_placeholder("请输入密码")

    assert expect_title.is_visible()
    assert expect_username.is_visible()
    assert expect_password.is_visible()
