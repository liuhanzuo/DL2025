import torch
import os
from torchvision.transforms import transforms
from dataset import CIFAR10_4x
from model_ import Net
import tqdm as tqdm
base_dir = os.path.dirname(__file__)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([125 / 255, 124 / 255, 115 / 255], [60 / 255, 59 / 255, 64 / 255])])


@torch.no_grad()
def evaluation(net, dataLoader, device):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in tqdm.tqdm(dataLoader, desc="Evaluating", leave=False):
            images, labels = data[0].to(device), data[1].to(device)
            outputs, _, _ = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the valid images: {accuracy:.2f}%')
    return accuracy


if __name__ == "__main__":
    bsz = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = Net().to(device)
    net = torch.load(os.path.join(base_dir, "checkpoint/cifar10_4x_best.pth"), weights_only=False)
    # print(net)
    print("number of trained parameters: %d" %
          (sum([param.nelement() for param in net.parameters() if param.requires_grad])))
    print("number of total parameters: %d" % (sum([param.nelement() for param in net.parameters()])))
    try:
        testset = CIFAR10_4x(root=base_dir, split='test', transform=transform)
    except Exception as e:
        testset = CIFAR10_4x(root=base_dir, split='valid', transform=transform)
        print("can't load test set because {}, load valid set now".format(e))
    testloader = torch.utils.data.DataLoader(testset, batch_size=bsz, shuffle=False, num_workers=2)

    evaluation(net, testloader, device)
