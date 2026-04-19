import asyncio
import io
import threading
import uuid

import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import DATABASE_URL
from app.core.security import hash_password
from app.models.user import User
from app.services import scan_service
from app.services.predict import predict

st.set_page_config(page_title="🍌 BananaTimer", layout="centered")


# ── async runner ──────────────────────────────────────────────────────────────

def run(coro_fn, *args, **kwargs):
    """Run an async function in a fresh thread with its own event loop and DB engine."""
    result = None
    exception = None

    def target():
        nonlocal result, exception
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Fresh engine per thread — avoids sharing asyncpg transports across loops
        engine = create_async_engine(DATABASE_URL, echo=False, pool_size=1, max_overflow=0)
        session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        try:
            result = loop.run_until_complete(coro_fn(*args, _sf=session_factory, **kwargs))
        except Exception as e:
            exception = e
        finally:
            loop.run_until_complete(engine.dispose())
            loop.close()

    t = threading.Thread(target=target)
    t.start()
    t.join()
    if exception:
        raise exception
    return result


# ── DB helpers ────────────────────────────────────────────────────────────────

async def _get_all_users(_sf) -> list[User]:
    async with _sf() as db:
        result = await db.execute(select(User))
        return list(result.scalars().all())


async def _save_scan(image_bytes: bytes, user_id: uuid.UUID, _sf=None):
    async with _sf() as db:
        return await scan_service.create_scan(image_bytes, user_id, db)


async def _get_history(user_id: uuid.UUID, _sf=None):
    async with _sf() as db:
        return await scan_service.predict_inedible_day(user_id, db)


async def _save_prehashed_user(email: str, hashed_password: str, full_name: str, _sf=None) -> User:
    async with _sf() as db:
        existing = await db.execute(select(User).where(User.email == email))
        if existing.scalar_one_or_none():
            return None  # double-submit guard
        user = User(email=email, hashed_password=hashed_password, full_name=full_name)
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user


# ── credentials loader ────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def _load_credentials() -> tuple[dict, dict]:
    """Load all users from DB and build the credentials dict for streamlit-authenticator."""
    users = run(_get_all_users)
    credentials: dict = {"usernames": {}}
    email_to_id: dict = {}
    for user in users:
        credentials["usernames"][user.email] = {
            "name": user.full_name or user.email.split("@")[0],
            "password": user.hashed_password,
        }
        email_to_id[user.email] = user.id
    return credentials, email_to_id


# ── authenticator setup ───────────────────────────────────────────────────────

credentials, email_to_id = _load_credentials()

authenticator = stauth.Authenticate(
    credentials,
    cookie_name="banana_timer_auth",
    cookie_key="banana_timer_super_secret",
    cookie_expiry_days=1,
    auto_hash=False,  # passwords are already hashed by our security module
)


# ── sidebar nav ───────────────────────────────────────────────────────────────

auth_status = st.session_state.get("authentication_status")

if auth_status:
    username = st.session_state.get("username")
    user_id = email_to_id.get(username)

    display_name = (credentials.get("usernames", {}).get(username) or {}).get("name") or username.split("@")[0]
    st.sidebar.success(f"👋 Welcome, **{display_name}**")
    authenticator.logout(button_name="Log out", location="sidebar")
    page = st.sidebar.radio("Navigation", ["🍌 Scan", "📈 History & Prediction"])

else:
    page = st.sidebar.radio("Go to", ["🔐 Login", "📝 Register"])


# ── Page: Login ───────────────────────────────────────────────────────────────

if not auth_status and page == "🔐 Login":
    st.title("🔐 Login")
    authenticator.login(location="main")

    if st.session_state.get("authentication_status") is False:
        st.error("Invalid email or password.")
    elif st.session_state.get("authentication_status") is None:
        st.info("Enter your credentials above.")


# ── Page: Register ────────────────────────────────────────────────────────────

elif not auth_status and page == "📝 Register":
    st.title("📝 Create Account")

    with st.form("register_form"):
        full_name = st.text_input("Full Name")
        reg_email = st.text_input("Email")
        reg_password = st.text_input("Password", type="password")
        reg_confirm = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register")

    if submitted:
        if not full_name or not reg_email or not reg_password:
            st.warning("Please fill in all fields.")
        elif reg_password != reg_confirm:
            st.error("Passwords do not match.")
        else:
            try:
                from app.core.security import hash_password as _hp
                hashed = _hp(reg_password)
                run(_save_prehashed_user, reg_email, hashed, full_name)
                # inject into in-memory credentials so authenticator can log them in
                credentials["usernames"][reg_email] = {
                    "name": full_name,
                    "password": hashed,
                }
                _load_credentials.clear()
                st.success("✅ Account created! Go to Login.")
            except ValueError as e:
                st.error(str(e))


# ── Page: Scan ────────────────────────────────────────────────────────────────

elif auth_status and page == "🍌 Scan":
    st.title("🍌 BananaTimer — Scan")
    st.write("Upload a banana photo to detect its ripeness stage.")

    uploaded_file = st.file_uploader("Upload banana image", type=["jpg", "jpeg", "png", "webp"])

    if st.button("Predict", use_container_width=True):
        if not uploaded_file:
            st.warning("Please upload an image.")
        else:
            image_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            with st.spinner("Analysing..."):
                result = predict(image)

            ripeness = result["ripeness"]
            days = result["days_until_inedible"]

            stage_emoji = {"unripe": "🟢", "ripe": "🟡", "overripe": "🟠", "rotten": "🔴"}
            st.success(f"{stage_emoji.get(ripeness, '')} **Ripeness:** {ripeness.capitalize()}")
            st.info(f"⏱ **Days estimate:** {days}")

            with st.spinner("Saving scan..."):
                try:
                    run(_save_scan, image_bytes, user_id)
                    st.caption("✅ Scan saved.")
                except Exception as e:
                    st.warning(f"Scan not saved: {e}")


# ── Page: History & Prediction ────────────────────────────────────────────────

elif auth_status and page == "📈 History & Prediction":
    st.title("📈 History & Prediction")

    if st.button("Load History", use_container_width=True):
        with st.spinner("Loading..."):
            try:
                data = run(_get_history, user_id)
            except ValueError as e:
                st.warning(str(e))
                st.stop()

        scans = data["scans"]
        df = pd.DataFrame(scans)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        st.subheader("Ripeness Progression")
        chart_df = df.set_index("date")[["stage"]].rename(columns={"stage": "Stage Index"})
        st.line_chart(chart_df)
        st.caption("Stage index: 1 = Overripe · 2 = Ripe · 3 = Rotten · 4 = Unripe")

        st.divider()
        st.subheader("Prediction")
        col1, col2 = st.columns(2)
        col1.metric("Days Left", f"{data['days_left']} days")
        col2.metric("Predicted Inedible Day", f"Day {data['predicted_inedible_day']}")

        st.subheader("Scan History")
        st.dataframe(
            df[["date", "ripeness", "stage"]].rename(
                columns={"date": "Date", "ripeness": "Ripeness", "stage": "Stage"}
            ),
            use_container_width=True,
        )
