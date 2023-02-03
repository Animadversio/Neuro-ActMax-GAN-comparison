from core.utils.layer_hook_utils import get_module_names, get_layer_names
import timm
model = timm.create_model('tf_efficientnet_b0_ap', pretrained=True)
model.eval()

module_names, module_types, module_spec = get_module_names(model, input_size=(3, 224, 224), device="cpu")