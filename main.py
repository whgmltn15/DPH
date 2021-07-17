from utils.config import parse_args
from utils.data_loader2 import get_data_loader
from model.model import nkModel
import torch
from utils.engine import train, val
from torch.utils.tensorboard import SummaryWriter

def main(args):

    summary = SummaryWriter(args.summary_location)
    train_loader, val_loader = get_data_loader(args)
    model = nkModel(args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.load_model: # 최솟값
        location = args.load_model_location
        print("load model: ", location)
        checkpoint = torch.load(location)
        model.load_state_dict(checkpoint['model_state_dict'])

    for epoch in range(args.start_epoch, args.max_epoch): # 몇번학습?

        train(model, train_loader, optimizer, epoch, summary) #학습
        val(model, val_loader, epoch, summary) #평가, optimizer 없는 이유 - 최적화 찾을 필요 x
        if epoch % 1 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       args.save_location + '/save_model/selectTF_model_{}.pth'.format(epoch))

if __name__ == '__main__':
    config = parse_args()
    main(config)