#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 09:42:24 2025
Dataclases for file menagement in adaptive sampling protocol
@author: igor
"""
import os, sys
from numpy import any
from typing import Literal

class FileBase():
    def __init__(self, fid:int, path:str, protected:bool=False):
        self.fid = fid
        self.abs_path = os.path.abspath(path)
        self.filename = os.path.basename(path)
        self.is_dir = os.path.isdir(path)
        self.ext = os.path.splitext(self.filename)
        self.protected = protected
        if self.protected:
            st = os.stat(self.abs_path)
            self.mtime = st.st_mtime
            self.size = st.st_size
        self.tags = set()
        
    def changed(self):
        if not self.protected: return False
        curr_st = os.stat(self.abs_path)
        if self.mtime != curr_st.mtime or self.size != curr_st.st_size:
            return True
        else:
            return False

class AdaptiveWorkplace(): #this system is weak when it comes to redundancy of file records, but i am too lazy to resolve it
    def __init__(self, root:str):
        if not os.path.exists(root):
            print(f"Given working directory location doesnt exist: {root}")
            sys.exit(1)
        self.root = FileBase(0,root)
        self.next_id = 1
        self.files = { 0 : self.root}
        self.tag_dict = {}
        
    def tag_file(self, fid:int, tags):
        if isinstance(tags, str):
            tags = [tags]
        for tag in tags:
            if tag not in self.tag_dict:
                self.tag_dict[tag] = set()
            self.tag_dict[tag].add(fid)
        self.files[fid].tags |= set(tags)
        
    def add_file(self, path:str, tags, protected:bool=False)->int:
        if not os.path.exists(path):
            print(f"Given path for file registration doesnt exist: {path}")
            sys.exit(1)
        fh = FileBase(self.next_id, path)
        self.next_id+=1
        self.files[fh.fid] = fh
        self.tag_file(fh.fid, tags)
        return fh.fid
    
    def unregister_file(self, fid:int, remove:bool=False):
        if fid not in self.files: 
            print(f"No file with ID: {fid} to remove")
            return 
        file_tags = self.files[fid].tags
        for tag in file_tags:
            self.tag_dict[tag].remove(fid)
        if remove:
            os.rmdir(self.files[fid].abs_path) if self.files[fid].is_dir else os.remove(self.files[fid].abs_path)
        self.files.pop(fid)
        
    def get_files_by_tags(self, tags, logic:Literal["or", "and", "contains"]="and"):
        if isinstance(tags, str):
            tags = [tags]
        selected_fids = set()
        if logic == "contains":
            if len(tags) > 1:
                print("Warning: mode \"contains\" works with only single tag")
                return set()
            for tag in self.tag_dict:
                if tags[0] in tag:
                    selected_fids |= self.tag_dict[tag]
        else:
            tmp = []
            for tag in tags:
                tmp.append(self.tag_dict[tag])
            if logic=="and":
                selected_fids = set.intersection(*tmp)
            elif logic=="or":
                selected_fids = set.union(*tmp)
        if len(selected_fids) == 1:
            selected_fids = list(selected_fids)[0]
        return selected_fids
    
    def get_files(self, fids):
        if isinstance(fids, int):
            return self.files[fids]
        else:
            return [self.files[fid] for fid in fids]

    def check_all_files(self)->bool:
        state = []
        for fid in self.files:
            state.append(self.files[fid].changed())
        return any(state)            
    
        
        
        
        
        
    
