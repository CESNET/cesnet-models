from typing import Optional

import numpy as np

from cesnet_models.architectures.multimodal_cesnet import Multimodal_CESNET, NormalizationEnum
from cesnet_models.helpers import Weights, WeightsEnum
from cesnet_models.transforms import ClipAndScaleFlowstats, ClipAndScalePPI, NormalizeHistograms


def _multimodal_cesnet(model_configuration: dict,
                       weights: Optional[WeightsEnum],
                       num_classes: Optional[int],
                       flowstats_input_size: Optional[int],
                       ppi_input_channels: Optional[int],
                       model_dir: Optional[str],) -> Multimodal_CESNET:
    if weights is None:
        if num_classes is None:
            raise ValueError("num_classes must be provided when weights are not provided")
        if flowstats_input_size is None:
            raise ValueError("flowstats_input_size must be provided when weights are not provided")
        if ppi_input_channels is None:
            raise ValueError("ppi_input_channels must be provided when weights are not provided")
    if weights is not None:
        if num_classes is not None and num_classes != weights.value.meta["num_classes"]:
            raise ValueError(f"Based on pre-trained weights, num_classes should be {weights.value.meta['num_classes']} but got {num_classes}")
        if flowstats_input_size is not None and flowstats_input_size != weights.value.meta["flowstats_input_size"]:
            raise ValueError(f"Based on pre-trained weights, flowstats_input_size should be {weights.value.meta['flowstats_input_size']} but got {flowstats_input_size}")
        if ppi_input_channels is not None and ppi_input_channels != weights.value.meta["ppi_input_channels"]:
            raise ValueError(f"Based on pre-trained weights, ppi_input_channels should be {weights.value.meta['ppi_input_channels']} but got {ppi_input_channels}")
        num_classes = weights.value.meta["num_classes"]
        flowstats_input_size = weights.value.meta["flowstats_input_size"]
        ppi_input_channels = weights.value.meta["ppi_input_channels"]
    assert num_classes is not None and flowstats_input_size is not None and ppi_input_channels is not None

    model = Multimodal_CESNET(**model_configuration, num_classes=num_classes, flowstats_input_size=flowstats_input_size, ppi_input_channels=ppi_input_channels)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(model_dir=model_dir))
    return model

class MM_CESNET_V2_Weights(WeightsEnum):
    CESNET_QUIC22_Week44 = Weights(
        bucket_url="https://liberouter.org/datazoo/download?bucket=cesnet-models",
        file_name="mmv2_CESNET_QUIC22_Week44.pth",
        transforms={
            "ppi_transform": ClipAndScalePPI(
                psizes_scaler_enum="standard",
                psizes_scaler_attrs={"mean_": [473.3267836449463], "scale_": [529.8389056483248]},
                pszies_min=1,
                psizes_max=1460,
                ipt_scaler_enum="standard",
                ipt_scaler_attrs={"mean_": [105.21803492421186], "scale_": [1061.4572842781342]},
                ipt_min=0,
                ipt_max=15000,),
            "flowstats_transform": ClipAndScaleFlowstats(
                flowstats_scaler_enum="robust",
                flowstats_scaler_attrs={
                    "center_": [4180.0, 5270.0, 12.0, 13.0, 0.2202340066432953, 25.0, 4.0, 0.15600000321865082],
                    "scale_": [4424.0, 7367.0, 14.0, 15.0, 4.964351028203964, 13.0, 3.0, 0.5059999749064445]},
                flowstats_quantiles=[135173.31999999983, 3184071.919999996, 539.0, 2663.0, np.inf, np.inf, np.inf, np.inf],
                quantile_clip=0.99,),
            "flowstats_phist_transform": NormalizeHistograms(),
        },
        meta={
            "train_dataset": "CESNET_QUIC22",
            "train_dataset_size": "ORIG",
            "train_period_name": "W-2022-44",
            "num_classes": 102,
            "classes": ["4chan", "adavoid", "alza-identity", "alza-webapi", "alza-www", "apple-privaterelay", "bitdefender-nimbus", "bitly", "blitz-gg", "blogger", "cedexis", "chess-com", "chrome-remotedesktop", "cloudflare-cdnjs", "connectad", "csgo-market", "dcard", "discord", "dm-de", "dns-doh", "doi-org", "drmax", "easybrain", "ebay-kleinanzeigen", "endnote-click", "etoro", "facebook-connect", "facebook-gamesgraph", "facebook-graph", "facebook-media", "facebook-messenger", "facebook-rupload", "facebook-web", "firebase-crashlytics", "fitbit", "flightradar24", "fontawesome", "forum24", "gamedock", "garmin", "gmail", "google-ads", "google-authentication", "google-autofill", "google-calendar", "google-colab", "google-conncheck", "google-docs", "google-drive", "google-fonts", "google-gstatic", "google-hangouts", "google-imasdk", "google-pay", "google-photos", "google-play", "google-recaptcha", "google-safebrowsing", "google-scholar", "google-services", "google-translate", "google-usercontent", "google-www", "goout", "hcaptcha", "hubspot", "instagram", "joinhoney", "jsdelivr", "kaggle", "kiwi-com", "livescore", "mdpi", "medium", "mentimeter", "microsoft-outlook", "microsoft-substrate", "ncbi-gov", "onesignal", "openx", "overleaf-cdn", "overleaf-compile", "playradio", "pocasidata-cz", "revolut", "rohlik", "shazam", "signal-cdn", "sme-sk", "snapchat", "spanbang", "spotify", "tawkto", "tiktok", "tinypass", "toggl", "uber", "unitygames", "usercentrics", "whatsapp", "xhamster", "youtube"],
            "ppi_input_channels": 3,
            "flowstats_input_size": 43,
            "use_packet_histograms": True,
            "num_params": 2_261_653,
            "paper_doi": "https://doi.org/10.23919/TMA58422.2023.10199052",
            "description": """These weights reproduce the results of the "Encrypted traffic classification: the QUIC case" paper."""
        }
    )
    DEFAULT = CESNET_QUIC22_Week44

def mm_cesnet_v2(weights: Optional[MM_CESNET_V2_Weights] = None,
                 model_dir: Optional[str] = None,
                 num_classes: Optional[int] = None,
                 flowstats_input_size: Optional[int] = None,
                 ppi_input_channels: Optional[int] = None,
                 ) -> Multimodal_CESNET:
    """
    This is a second version of the multimodal CESNET architecture. It was used in
    the *"Encrypted traffic classification: the QUIC case"* paper.

    Changes from the first version:
        - Global pooling was added to the CNN part processing PPI sequences, instead of a simple flattening.
        - One more Conv1d layer was added to the CNN part and the number of channels was increased.
        - The size of the MLP processing flow statistics was increased.
        - The size of the MLP processing shared representations was decreased.
        - Some dropout rates were decreased.

    Parameters:
        weights: If provided, the model will be initialized with these weights.
        model_dir: If weights are provided, this folder will be used to store the weights.
        num_classes: Number of classes for the classification task.
        flowstats_input_size: Size of the flow statistics input.
        ppi_input_channels: Number of channels in the PPI input.
    """
    v2_model_configuration = {
        "conv_normalization": NormalizationEnum.BATCH_NORM,
        "linear_normalization": NormalizationEnum.BATCH_NORM,
        "cnn_num_hidden": 3,
        "cnn_channels1": 200,
        "cnn_channels2": 300,
        "cnn_channels3": 300,
        "cnn_use_pooling": True,
        "cnn_dropout_rate": 0.1,
        "flowstats_num_hidden": 2,
        "flowstats_size": 225,
        "flowstats_out_size": 225,
        "flowstats_dropout_rate": 0.1,
        "latent_num_hidden":  0,
        "latent_size": 600,
        "latent_dropout_rate": 0.2,
    }
    return _multimodal_cesnet(model_configuration=v2_model_configuration,
                              weights=weights,
                              model_dir=model_dir,
                              num_classes=num_classes,
                              flowstats_input_size=flowstats_input_size,
                              ppi_input_channels=ppi_input_channels)

class MM_CESNET_V1_Weights(WeightsEnum):
    CESNET_TLS22_WEEK40 = Weights(
        bucket_url="https://liberouter.org/datazoo/download?bucket=cesnet-models",
        file_name="mmv1_CESNET_TLS22_WEEK40.pth",
        transforms={
            "ppi_transform": ClipAndScalePPI(
                psizes_scaler_enum="standard",
                psizes_scaler_attrs={"mean_": [708.5765387201488],"scale_": [581.2818120048021]},
                pszies_min=1,
                psizes_max=1460,
                ipt_scaler_enum="standard",
                ipt_scaler_attrs={"mean_": [228.1051793224929],"scale_": [1517.0445763930045]},
                ipt_min=1,
                ipt_max=15000,),
            "flowstats_transform": ClipAndScaleFlowstats(
                flowstats_scaler_enum="robust",
                flowstats_scaler_attrs={
                    "center_": [2494.0, 6362.0, 13.0, 13.0, 0.5721004903316498, 13.0, 3.0, 0.20399999618530273],
                    "scale_": [3028.0, 5321.0, 9.0, 10.0, 5.809830829501152, 11.0, 2.0, 0.7719999849796295]},
                flowstats_quantiles=[19151.0, 125462.04999999888, 79.0, 126.0, np.inf, np.inf, np.inf, np.inf],
                quantile_clip=0.95,)
        },
        meta={
            "train_dataset": "CESNET_TLS22",
            "train_dataset_size": "ORIG",
            "train_period_name": "W-2021-40",
            "num_classes": 191,
            "classes": ["3dsecure", "accuweather", "adobe-ads", "adobe-analytics", "adobe-authentication", "adobe-cloud", "adobe-notifications", "adobe-search", "adobe-updater", "airbank-ib", "alza-bnr", "alza-cdn", "alza-logapi", "alza-signalr", "alza-webapi", "amazon-advertising", "amazon-alexa", "amazon-prime", "apple-icloud", "apple-itunes", "apple-location", "apple-ocsp", "apple-pancake", "apple-push", "apple-updates", "apple-weather", "appnexus", "aukro-backend", "autodesk", "avast", "bing", "bitdefender-gravityzone", "bitdefender-nimbus", "booking-com", "cesnet-filesender", "cesnet-gerrit", "cesnet-kalendar", "cesnet-login", "cesnet-nerd", "cesnet-perun", "chmi", "chrome-remotedesktop", "csas-webchat", "ctu-felmail", "ctu-idp2", "ctu-kos", "ctu-kosapi", "ctu-matrix", "datova-schranka", "discord", "dns-doh", "docker-auth", "docker-registry", "dopravniinfo-api", "dropbox", "duckduckgo", "ea-games", "edge-ntp", "eidentita", "ekasa", "eset-edf", "eset-edtd", "eset-epns", "eset-esa", "eset-ts", "facebook-graph", "facebook-media", "facebook-messenger", "facebook-web", "fio-ib", "firefox-accounts", "firefox-settings", "font-awesome", "gfe-events", "gfe-services", "github", "gitlab", "gmail", "google-ads", "google-authentication", "google-connectivity", "google-drive", "google-fonts", "google-hangouts", "google-play", "google-safebrowsing", "google-services", "google-translate", "google-userlocation", "google-www", "grammarly", "hicloud-connectivity", "hicloud-logservice", "instagram", "justice-isir", "kaspersky", "katastr-nahlizeni", "kb-ib", "king-games", "loggly", "malwarebytes-telemetry", "mapscz", "mcafee-ccs", "mcafee-gti", "mcafee-realprotect", "microsoft-authentication", "microsoft-defender", "microsoft-diagnostic", "microsoft-notes", "microsoft-onedrive", "microsoft-push", "microsoft-settings", "microsoft-update", "microsoft-weather", "mlp-search", "moneta-ib", "moodle", "mozilla-location", "mozilla-push", "mozilla-telemetry", "mozilla-token", "ndk", "netflix", "npm-registry", "o2tv", "obalkyknih", "office-365", "opera-autoupdate", "opera-notifications", "opera-oauth2", "opera-sitecheck", "opera-speeddial", "opera-weather", "outlook", "owncloud", "pubmatic", "r2d2", "rb-ib", "redmine", "riot-games", "rozhlas-api", "rubiconproject", "salesforce", "seznam-authentication", "seznam-email", "seznam-media", "seznam-notifications", "seznam-search", "seznam-ssp", "signageos", "skype", "slack", "snapchat", "soundcloud", "spotify", "steam", "sukl-api", "sukl-auth", "sukl-erecept", "sumava-camdata", "super-media", "teams", "teamviewer-client", "the-weather-channel", "thunderbird-telemetry", "tiktok", "tinder", "twitch", "twitter", "ulozto", "unity-games", "unpkg", "uschovna", "uzis-api", "uzis-ocko", "uzis-plf", "vimeo", "visualstudio-insights", "vmware-vcsa", "vsb-sso", "vscode-update", "vse-insis", "vzp-api", "webex", "whatsapp", "xbox-live", "xiaomi-tracking", "yahoo-mail", "youtube", "zoom", "zotero"],
            "ppi_input_channels": 3,
            "flowstats_input_size": 17,
            "use_tcp_features": True,
            "num_params": 1_217_903,
            "paper_doi": "https://doi.org/10.1016/j.comnet.2022.109467",
            "description":  """These weights reproduce the results of the "Fine-grained TLS services classification with reject option" paper."""
        }
    )
    DEFAULT = CESNET_TLS22_WEEK40

def mm_cesnet_v1(weights: Optional[MM_CESNET_V1_Weights] = None,
                 model_dir: Optional[str] = None,
                 num_classes: Optional[int] = None,
                 flowstats_input_size: Optional[int] = None,
                 ppi_input_channels: Optional[int] = None,
                 ) -> Multimodal_CESNET:
    """
    This model was used in the *"Fine-grained TLS services classification with reject option"* paper.

    Parameters:
        weights: If provided, the model will be initialized with these weights.
        model_dir: If weights are provided, this folder will be used to store the weights.
        num_classes: Number of classes for the classification task.
        flowstats_input_size: Size of the flow statistics input.
        ppi_input_channels: Number of channels in the PPI input.
    """
    v1_model_configuration = {
        "conv_normalization": NormalizationEnum.BATCH_NORM,
        "linear_normalization": NormalizationEnum.BATCH_NORM,
        "cnn_num_hidden": 2,
        "cnn_channels1": 72,
        "cnn_channels2": 128,
        "cnn_channels3": 128,
        "cnn_use_pooling": False,
        "cnn_dropout_rate": 0.2,
        "flowstats_num_hidden": 2,
        "flowstats_size": 64,
        "flowstats_out_size": 32,
        "flowstats_dropout_rate": 0.2,
        "latent_num_hidden": 1,
        "latent_size": 480,
        "latent_dropout_rate": 0.2,
    }
    return _multimodal_cesnet(model_configuration=v1_model_configuration,
                              weights=weights,
                              model_dir=model_dir,
                              num_classes=num_classes,
                              flowstats_input_size=flowstats_input_size,
                              ppi_input_channels=ppi_input_channels)
