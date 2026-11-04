from product import runtime

runtime.prefer_offline_models()

from product.ui.app import main  # noqa: E402

raise SystemExit(main())
